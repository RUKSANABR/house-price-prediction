import React, { useState } from 'react';
import { View, Text, StyleSheet, TouchableOpacity, ScrollView } from 'react-native';
import { Avatar } from '@rneui/base';
import ContracterAccepted from '../components/contracter/ContracterAccepted';
import ContracterAll from '../components/contracter/ContracterAll';
import ContracterDeclined from '../components/contracter/ContracterDeclined';

export default function ContracterPage({ navigation, route }) {
  const { userD } = route.params;
  const [selectedItem, setSelectedItem] = useState('All');

  const handleItemClick = (itemName) => {
    setSelectedItem(itemName);
  };


  return (
    <View style={styles.container}>
      <View style={styles.s1}>
        <Text style={styles.mainText}>Hello {userD.name}</Text>
        <Avatar
          source={require('../assets/bell.png')}
          onPress={() => navigation.navigate('Notification', { userD })}
        />
        <Avatar
          source={require('../assets/person.png')}
          containerStyle={{ marginRight: 10 }}
          onPress={() => navigation.navigate('Profile', { userD })}
        />
      </View>
      <ScrollView>
        <Text style={[styles.mainText, { marginTop: 20, padding: 10 }]}>
          Contracter
        </Text>

        <View style={[styles.s1, { justifyContent: 'flex-start' }]}>
          <TouchableOpacity
            style={[
              styles.panel,
              { backgroundColor: selectedItem === 'All' ? '#e1e3da' : 'white' },
            ]}
            onPress={() => handleItemClick('All')}
          >
            <Text style={styles.panelText}>All</Text>
          </TouchableOpacity>
          <TouchableOpacity
            style={[
              styles.panel,
              {
                backgroundColor: selectedItem === 'Accepted' ? '#e1e3da' : 'white',
              },
            ]}
            onPress={() => handleItemClick('Accepted')}
          >
            <Text style={styles.panelText}>Accepted</Text>
          </TouchableOpacity>
          <TouchableOpacity
            style={[
              styles.panel,
              {
                backgroundColor: selectedItem === 'Declined' ? '#e1e3da' : 'white',
              },
            ]}
            onPress={() => handleItemClick('Declined')}
          >
            <Text style={styles.panelText}>Declined</Text>
          </TouchableOpacity>
        </View>

        {/* Render the components based on the selected category */}
        {selectedItem === 'All' && <ContracterAll userD={userD} />}
        
        {selectedItem === 'Accepted' && <ContracterAccepted userD={userD} />}
        
        {selectedItem === 'Declined' && <ContracterDeclined userD={userD} />}
      </ScrollView>
    </View>
  );
}
styles = StyleSheet.create({
  container: {
    // backgroundColor:'black'
    marginBottom:40
  },
  s1: {
    padding: 10,
    justifyContent: 'space-between',
    flexDirection: 'row',
    marginTop: 10,
  },
  mainText: {
    fontSize: 25,
    color: 'black',
    fontWeight: '500',
  },
  panel: {
    width: 110,
    height: 40,
    backgroundColor: 'white',
    borderRadius: 20,
    margin: 10,
    alignItems: 'center', // Center horizontally
    justifyContent: 'center',
  },
  panelText: {
    textAlign: 'center',
    fontSize: 20,
    fontWeight: '500',
    color: 'black',
  },
  
});
